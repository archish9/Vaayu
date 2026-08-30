"""FastAPI surface. Routes, error mapping and SSE framing only -- no policy.

Everything decidable without a GPU lives in stopping.py / tools.py / schemas.py.
"""
import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import NoChatTemplate, TemplateLacksToolSupport
from .engine import CancelToken, EngineHandle, QueueFull
from .logs import capture_stdio
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
        try:
            if config is not None and config.capture_stdio:
                # Before the load, so pinning and NVMe warmup end up in the file too.
                restore_stdio = capture_stdio(
                    Path(config.log_dir) / "server.log", rotate_mb=config.log_rotate_mb)
                print(f"[Vaayu-Server] logging to {config.log_dir}/server.log and "
                      f"{config.log_dir}/sessions/<session_id>.log")
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
                    attn_implementation=config.attn_implementation,
                    ram_cache_budget=config.ram_cache_budget,
                    prefix_sessions=config.prefix_sessions,
                )
                print(f"[Vaayu-Server] loaded in {time.time() - t0:.1f}s; "
                      f"derived max_seq_len={model.max_seq_len}")
                app.state.engine = EngineHandle(model, queue_depth=config.queue_depth,
                                                enable_thinking=config.enable_thinking,
                                                log_dir=config.log_dir)
                # Only on the path that actually loaded: an injected engine
                # belongs to a caller that keeps using this process, and
                # freezing its objects would retain them for the process's life.
                freeze_after_load()
            else:
                app.state.engine = engine
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
            _teardown(app)
            if restore_stdio is not None:
                restore_stdio()

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
        cfg = getattr(app.state, "config", None)
        if cfg is not None and cfg.require_session_id and not req.session_key:
            return _error(400,
                          "this request names no session. Send `session_id` (or "
                          "OpenAI's `user`, which is used as one). Without it the "
                          "prefix cache keys on a hash of the prompt's first 64 "
                          "tokens, so two agents sharing a system prompt collide "
                          "on one entry and evict each other -- reuse then "
                          "collapses to zero rather than degrading. Set "
                          "REQUIRE_SESSION_ID = False to allow unnamed requests.",
                          "invalid_request_error")
        warning = e.warn_if_speculative_sampling_ignored(req)
        if warning:
            print(f"[Vaayu-Server] WARNING: {warning}")

        if resolve_stream(req, getattr(app.state, "config", None)):
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
            try:
                # Build the prompt HERE rather than inside the generator. A
                # StreamingResponse's body runs after this handler returns, so
                # anything raised there can only be a 200 carrying an SSE error
                # event -- and DEFAULT_STREAM is True, so an oversized prompt
                # reported "internal_error" to every default client (B8). The
                # result is handed to _sse so it is not built twice.
                prepared = e.prepare(req)
            except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
                return _error(400, str(ex), "invalid_request_error")
            return StreamingResponse(
                _sse(e, req, _served_name(app), warning, CancelToken(),
                     session_id=req.session_key, prepared=prepared),
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
               prepared=None):
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

    yield _chunk(cid, created, model_name, session_id=session_id, delta={"role": "assistant"})

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
    try:
        # Acquired HERE, not in the route: this generator body runs after the
        # handler returned, so a slot held around that return would already be
        # released before the first token.
        async with engine.slot():
            async for kind, payload in engine.stream(req, cancel=cancel,
                                                     prepared=prepared):
                if kind == "final":
                    final = payload
                    break
                if suppressing:
                    buffered += payload
                    continue
                held += payload
                if open_marker and open_marker in held:
                    # Everything from here MIGHT belong to a tool call. Emit the
                    # prose that preceded it and buffer the rest until the final
                    # event says which it was.
                    cut = held.index(open_marker)
                    if held[:cut]:
                        sent += held[:cut]
                        yield _chunk(cid, created, model_name, session_id=session_id,
                                     delta={"content": held[:cut]})
                    held, buffered, suppressing = "", held[cut:], True
                    continue
                # Hold back a marker-length tail so an open marker split across
                # two chunks is still recognised.
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
        yield _error_event(ex, "server_busy")
        yield "data: [DONE]\n\n"
        return
    except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
        # Unreachable from the route, which now prepares before responding.
        # Kept for any other caller of _sse, and so this class of error is
        # never again reported as a fault of the server's.
        yield _error_event(ex, "invalid_request_error")
        yield "data: [DONE]\n\n"
        return
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
