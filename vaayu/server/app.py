"""FastAPI surface. Routes, error mapping and SSE framing only -- no policy.

Everything decidable without a GPU lives in stopping.py / tools.py / schemas.py.
"""
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import NoChatTemplate, TemplateLacksToolSupport
from .engine import EngineHandle, QueueFull
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


def _error(status: int, message: str, kind: str, headers=None):
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": kind}})


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
            )
            print(f"[Vaayu-Server] loaded in {time.time() - t0:.1f}s; "
                  f"derived max_seq_len={model.max_seq_len}")
            app.state.engine = EngineHandle(model, queue_depth=config.queue_depth)
        else:
            app.state.engine = engine
        yield

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
    async def chat_completions(req: ChatCompletionRequest):
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
                _sse(e, req, _served_name(app), warning),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            async with e.slot():
                result = await e.generate(req)
        except QueueFull as ex:
            return _error(503, str(ex), "server_busy", headers={"Retry-After": "5"})
        except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
            return _error(400, str(ex), "invalid_request_error")
        except Exception as ex:                       # noqa: BLE001
            return _error(500, f"{type(ex).__name__}: {ex}", "internal_error")

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
        return body

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


async def _sse(engine, req, model_name, warning):
    """OpenAI-shaped SSE. Content streams; a tool call is BUFFERED and emitted
    once, complete -- partial tool-call JSON breaks clients that parse each
    delta, and the gain from streaming it is nil."""
    cid, created = make_completion_id(), int(time.time())

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
            async for kind, payload in engine.stream(req):
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
