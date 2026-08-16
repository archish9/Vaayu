"""FastAPI surface. Routes, error mapping and SSE framing only -- no policy.

Everything decidable without a GPU lives in stopping.py / tools.py / schemas.py.
"""
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import NoChatTemplate, TemplateLacksToolSupport
from .engine import EngineHandle, QueueFull
from .schemas import (ChatCompletionRequest, ChatCompletionResponse, Choice,
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


async def _sse(engine, req, model_name, warning):
    raise NotImplementedError("streaming lands in Task 9")
    yield  # pragma: no cover -- makes this an async generator
