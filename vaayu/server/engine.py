"""EngineHandle -- the one loaded model, the admission gate, and generation.

Concurrency is 1 by construction. The engine is not thread-safe, and N>=2 at
long context is a settled hardware refusal on this box (peak 14.17-15.57 GiB of
15.47), so this is a depth-bounded FIFO gate, not a scheduler. `batched_generate`
is deliberately not wired in: it is static-batch v1 and would admit nothing here.
"""
import asyncio
import queue as _queue
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import reqlog
from .chat import render_prompt
from .logs import clear_current_session, session_event, set_current_session
from .stopping import finish_reason, resolve_max_tokens
from .tools import declared_tool_names, detect_tool_format, parse_tool_calls


# Keyword arguments vaayu.streaming.stream_generate constructs itself. Anything
# here that also arrives via **kwargs is a duplicate-keyword TypeError. Kept as a
# named set so a change to stream_generate's signature has one place to land.
STREAM_GENERATE_OWNS = ("input_ids", "streamer", "max_new_tokens", "temperature",
                        "top_p", "do_sample", "use_cache")


class QueueFull(RuntimeError):
    """More requests are waiting than the configured depth allows."""


class CancelToken:
    """Set from the event loop, read from the generation thread.

    A blocking generate() in an executor thread cannot be killed from outside,
    so cancellation is cooperative: this flag is checked once per token. Without
    it a client that walks away leaves generation running -- measured holding
    the GPU and the single slot with no connection left on the port.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def cancel_criteria(token: CancelToken):
    """A transformers StoppingCriteriaList that stops when `token` is cancelled.

    Imports live here rather than at module scope so `import vaayu.server` stays
    as cheap as it was. The criteria must answer per batch ROW -- StoppingCriteria
    List ORs the results into a (batch,) tensor, and a bare bool works right up
    until something batches.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), token.cancelled,
                              dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_Cancelled()])


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    tool_calls: list = field(default_factory=list)


class EngineHandle:
    def __init__(self, model, queue_depth: int = 8, enable_thinking: bool = True,
                 log_dir=None):
        self.model = model
        self.queue_depth = queue_depth
        # None for library callers: a library that starts writing files into the
        # caller's working directory is a surprise. The server passes its own.
        self.log_dir = log_dir
        # Library default stays True (render exactly what the template wrote);
        # the server passes its own, which is False.
        self.enable_thinking = enable_thinking
        self._lock = asyncio.Lock()
        self._inflight = 0
        # The token of the request currently generating, so shutdown can stop
        # it. Concurrency is 1 by construction, so one slot is the whole set.
        self._live_cancel = None
        # The worker thread that token belongs to. The token alone cannot say
        # whether generation has actually STOPPED -- cancellation is cooperative
        # and is only checked once per token -- so `quiesce` needs the thread.
        self._live_thread = None

        # vaayu_base falls back to max_seq_len=512 when it built no VramPlan
        # (CPU / no CUDA). This server defaults the generation cap to "the
        # remaining window", so that fallback would silently reinstate exactly
        # the fixed 512-token cap the server exists to remove. Name it loudly.
        if getattr(model, "_vram_plan", None) is None:
            print(f"[Vaayu-Server] WARNING: this model has no VRAM plan (CPU or "
                  f"no CUDA), so max_seq_len is vaayu_base's fallback of "
                  f"{self.max_seq_len}. Every request's default token cap is "
                  f"bounded by it. Load on CUDA for a derived window.")

    # ── properties read off the model, never cached ──────────────────────────

    @property
    def max_seq_len(self) -> int:
        """Derived at load by plan_vram; never a constant in this package."""
        return self.model.max_seq_len

    @property
    def tool_format(self):
        return detect_tool_format(getattr(self.model.tokenizer, "chat_template", None))

    @property
    def depth(self) -> int:
        return self._inflight

    @property
    def live_cancel(self):
        """The running request's CancelToken, or None when idle."""
        return self._live_cancel

    def cancel_all(self) -> None:
        """Stop whatever is generating. Idempotent, and a no-op when idle."""
        token = self._live_cancel
        if token is not None:
            token.cancel()

    def quiesce(self) -> None:
        """Cancel the running generation and WAIT for its worker to exit.

        Deliberately synchronous. It runs on the teardown path of a cancelled
        request, where `await` is not dependable: the ASGI server cancels the
        task, so an anyio cancel scope re-raises CancelledError at every
        checkpoint and the wait would be skipped -- which is the whole bug. A
        blocking join always runs. It costs at most the one forward pass the
        cancel flag is checked after, and the GPU is busy for that time anyway.
        """
        self.cancel_all()
        thread = self._live_thread
        if thread is not None:
            thread.join()
            self._live_thread = None

    def prefix_stats(self) -> str:
        """One grep-able line from the prefix pool, or a note that it is off.

        Hit rate here is the difference between ~1 s and ~98 s TTFT, and there
        is otherwise no way to observe it from outside the process.
        """
        pc = getattr(self.model, "_prefix_cache", None)
        if pc is None:
            return "disabled"
        try:
            return pc.stats()
        except Exception as e:
            return f"unavailable ({type(e).__name__}: {e})"

    # ── admission ────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def slot(self, session_key=None):
        """Admit one request. Raises QueueFull past `queue_depth`.

        The counter is incremented BEFORE awaiting the lock, so it counts
        waiting requests as well as the running one -- a client's queue position
        is `depth` at entry.
        """
        if self._inflight >= self.queue_depth:
            raise QueueFull(
                f"{self._inflight} requests are already queued (depth "
                f"{self.queue_depth}). This engine runs one request at a time.")
        self._inflight += 1
        try:
            async with self._lock:
                # INSIDE the lock, not before it: that is what makes the
                # marker's overlap check assert the real invariant (one
                # generation at a time) rather than something about the
                # admission queue.
                set_current_session(session_key)
                try:
                    yield
                finally:
                    # The slot must outlive the generation on EVERY exit path.
                    # A client disconnect closes the SSE generator with
                    # GeneratorExit, which unwinds through here while
                    # stream_generate's worker is still inside model.generate;
                    # releasing the lock now admits a second generate() onto an
                    # engine that is not thread-safe and shares the KV cache,
                    # the expert VRAM cache and the slab arena.
                    try:
                        self.quiesce()
                    finally:
                        # After quiesce: lines emitted while the worker is
                        # being joined still belong to this session.
                        clear_current_session()
        finally:
            self._inflight -= 1

    # ── request mapping ──────────────────────────────────────────────────────

    def _log(self, req, event, **fields) -> None:
        if self.log_dir is not None:
            session_event(self.log_dir, getattr(req, "session_key", None),
                          event, **fields)

    def _write_request_row(self, req, *, stream: bool, started: float,
                           prompt_tokens: int, cap: int, snap_before: dict,
                           tokens: int = 0, finish: str = "error",
                           ttft_s=None, hits_before: int = 0, error=None) -> None:
        """One `requests.jsonl` row. Called from a `finally`, so it runs on the
        success path, the raise path, and the client-disconnect path alike --
        which is the point: audit item 9 was that a failed request left a REQ
        line with no DONE and an unattributed traceback in server.log.

        Never raises (reqlog.write_row swallows), and never computes anything
        it cannot: a request that produced no tokens gets None per-token means
        rather than a ZeroDivisionError on top of whatever already failed.

        The whole body is additionally wrapped in its own try/except: this
        runs from a `finally` on the generation path, and an exception raised
        HERE -- reqlog.snapshot, decode_block, error_block, or the dict
        literal itself, none of which `reqlog.write_row`'s own swallow covers
        -- would replace whatever real exception was already propagating (a
        CUDA OOM becoming a confusing logging traceback instead). A lost log
        line is cheaper than a lost diagnosis of the thing that actually failed.
        """
        import time
        try:
            wall = time.time() - started
            snap_after = reqlog.snapshot(self.model)
            reqlog.write_row(self.log_dir, {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "session": getattr(req, "session_key", None),
                "prompt_tokens": prompt_tokens,
                "cap": cap,
                "stream": stream,
                "thinking": self.resolve_thinking(req),
                "max_tokens": getattr(req, "max_tokens", None),
                "temperature": getattr(req, "temperature", None),
                "tools": len(getattr(req, "tools", None) or []),
                "prefix_hit": self._prefix_hits() > hits_before,
                "ttft_s": ttft_s,
                "wall_s": wall,
                "tokens": tokens,
                "finish": finish,
                "decode_tok_s": ((tokens - 1) / wall
                                 if wall > 0 and tokens > 1 else None),
                "decode": reqlog.decode_block(snap_before, snap_after, tokens),
                "error": reqlog.error_block(error),
            })
        except Exception:                            # noqa: BLE001 -- never displaces
            pass

    def _prefix_hits(self) -> int:
        pc = getattr(self.model, "_prefix_cache", None)
        return getattr(pc, "hits", 0) if pc is not None else 0

    def resolve_thinking(self, req) -> bool:
        """The request's choice if it made one, else the server's."""
        if getattr(req, "enable_thinking", None) is not None:
            return req.enable_thinking
        return self.enable_thinking

    def prepare(self, req):
        """(prompt_ids, resolved_cap). Raises ContextTooLong / TemplateLacksToolSupport."""
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        tools = [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
        prompt_ids = render_prompt(self.model.tokenizer, messages, tools=tools,
                                   enable_thinking=self.resolve_thinking(req))
        cap = resolve_max_tokens(req.max_tokens, len(prompt_ids), self.max_seq_len)
        return prompt_ids, cap

    def warn_if_speculative_sampling_ignored(self, req):
        """speculative_generate is GREEDY -- temperature included, though it is a
        declared parameter and this docstring used to claim it was honoured.
        Return a message when this request asks for something that path will
        drop, else None.

        Asked of the same helper the engine warns with, against the kwargs this
        request would actually be run with, so the two can never disagree.
        """
        from ..vaayu_base import speculative_ignored_sampling

        if getattr(self.model, "speculator", None) is None:
            return None
        kwargs = {}
        if req.temperature > 0.0:
            kwargs.update(do_sample=True, top_p=req.top_p, top_k=req.top_k)
        if req.stop:
            kwargs["stop_strings"] = list(req.stop)
        ignored = speculative_ignored_sampling(req.temperature, kwargs)
        if not ignored:
            return None
        return (f"speculative decoding is active and its loop is greedy, so "
                f"{', '.join(ignored)} will be IGNORED for this request. "
                f"Load without drafter_model_path to use them.")

    def build_generate_kwargs(self, req, prompt_ids, cap=None, cancel=None) -> dict:
        import torch
        kwargs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long,
                                      device=self.model.running_device),
            "max_new_tokens": cap if cap is not None else 1,
            "use_cache": True,
            # session_id, else OpenAI's `user`, else None -> auto-prefix-match.
            "session_id": req.session_key,
        }
        if req.temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = req.temperature
            kwargs["top_p"] = req.top_p
            kwargs["top_k"] = req.top_k
        else:
            kwargs["do_sample"] = False

        stops = list(req.stop)
        fmt = self.tool_format
        if req.tools and fmt is not None:
            # Halt AT the call rather than running on to hallucinate its result.
            for s in fmt.stop_strings():
                if s not in stops:
                    stops.append(s)
        if stops:
            kwargs["stop_strings"] = stops
            kwargs["tokenizer"] = self.model.tokenizer   # HF requires it with stop_strings
        if cancel is not None:
            # Honoured by transformers' loop AND by speculative_generate, which
            # has no criteria machinery of its own and checks this per round.
            kwargs["stopping_criteria"] = cancel_criteria(cancel)
        return kwargs

    @asynccontextmanager
    async def _running(self, cancel, thread=None):
        """Publish this request's token -- and its worker -- for the lifetime of
        the generation, and do not clear either until that worker has EXITED.

        Same contract as `stream()`'s teardown (B9), for the non-streaming route
        (B16). Clearing on the way out of a raising `await` leaves `cancel_all()`
        reporting idle with a generation still on the GPU, and leaves
        `slot()`'s `quiesce()` with no thread to join -- so the lock is released
        under a live generation. The cancel must PRECEDE the join: an
        uncancelled join on the abandoned path would block for the whole
        remaining cap.
        """
        self._live_cancel = cancel
        self._live_thread = thread
        try:
            yield
        except BaseException:                   # noqa: BLE001 -- re-raised
            # The handler task was cancelled (uvicorn's graceful-shutdown
            # timeout, lifespan teardown) while the worker is still inside
            # model.generate.
            cancel.cancel()
            raise
        finally:
            try:
                if thread is not None:
                    thread.join()
            finally:
                self._live_cancel = None
                self._live_thread = None

    # ── generation ───────────────────────────────────────────────────────────

    def _decode_new(self, out_ids, prompt_len):
        new_ids = out_ids[0][prompt_len:].tolist()
        text = self.model.tokenizer.decode(new_ids, skip_special_tokens=True)
        return new_ids, text

    def _finish(self, new_ids, cap, tool_calls, cancel=None):
        """finish_reason for a run, from its real generated ids.

        Shared by both paths. `new_ids` must be the ids the model actually
        generated -- the streaming path used to re-tokenize its own decoded
        output here, which does not reproduce them.
        """
        eos = self.model.config.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        return finish_reason(
            hit_eos=bool(new_ids) and new_ids[-1] in eos_set,
            # HF trims a matched stop string before returning, so the returned
            # ids cannot be inspected for one. `produced < cap` covers it: the
            # only distinction a client acts on is length vs stop. A CANCELLED
            # run also satisfies `produced < cap`, which is why the token is
            # passed too -- finish_reason consults it before this inference.
            hit_stop_string=len(new_ids) < cap,
            hit_tool_call=bool(tool_calls),
            produced=len(new_ids), cap=cap,
            cancelled=cancel is not None and cancel.cancelled)

    async def generate(self, req, cancel=None) -> GenerationResult:
        import time
        if cancel is None:
            # Always publish a REAL token, for the same reason stream() does
            # (B9): `cancel_all()` reads it to answer "is anything generating?",
            # and the teardown below has to have something to cancel before it
            # waits for the worker (B16).
            cancel = CancelToken()
        started, hits_before = time.time(), self._prefix_hits()
        snap_before = reqlog.snapshot(self.model)
        prompt_ids, cap = self.prepare(req)
        self._log(req, "REQ", prompt=len(prompt_ids), cap=cap, stream=False,
                  thinking=("on" if self.resolve_thinking(req) else "off"))
        if req.max_tokens is None:
            # Non-streaming sends NOTHING until the last token, so this request
            # can sit silent for as long as `cap` tokens take -- measured at 15+
            # minutes and counting on MiniMax-M2.7 before a client gave up and
            # the run looked hung. It is not hung; it is uncapped.
            print(f"[Vaayu-Server] WARNING: non-streaming request with no "
                  f"max_tokens: up to {cap} tokens will be generated before ANY "
                  f"bytes are sent. Set max_tokens, or use stream=true.")
        row = {"tokens": 0, "finish": "error", "error": None}
        try:
            kwargs = self.build_generate_kwargs(req, prompt_ids, cap, cancel=cancel)
            loop = asyncio.get_running_loop()
            # An OWN thread, not a run_in_executor pool worker: `quiesce()` joins
            # what is published here, and a pool thread never exits, so joining one
            # would hang forever. The bridge mirrors stream()'s pump.
            bridge: _queue.Queue = _queue.Queue()

            def _work():
                try:
                    bridge.put(("out", self.model.generate(**kwargs)))
                except BaseException as ex:         # noqa: BLE001 -- re-raised below
                    bridge.put(("error", ex))

            thread = threading.Thread(target=_work, daemon=True)
            async with self._running(cancel, thread):
                thread.start()
                kind, payload = await loop.run_in_executor(None, bridge.get)
            if kind == "error":
                raise payload
            out = payload
            new_ids, text = self._decode_new(out, len(prompt_ids))
            cleaned, calls = parse_tool_calls(text, self.tool_format if req.tools else None,
                                      declared_tool_names(req.tools))
            finish = self._finish(new_ids, cap, calls, cancel=cancel)
            wall = time.time() - started
            self._log(req, "DONE", tokens=len(new_ids), finish=finish,
                      prefix_hit=self._prefix_hits() > hits_before,
                      wall=f"{wall:.1f}s",
                      decode=f"{(len(new_ids) - 1) / wall:.3f}tok/s" if wall > 0 and len(new_ids) > 1 else "n/a")
            row["tokens"], row["finish"] = len(new_ids), finish
            return GenerationResult(
                text=cleaned, tool_calls=calls,
                prompt_tokens=len(prompt_ids), completion_tokens=len(new_ids),
                finish_reason=finish)
        except BaseException as ex:              # noqa: BLE001 -- re-raised
            row["error"] = ex
            raise
        finally:
            self._write_request_row(
                req, stream=False, started=started,
                prompt_tokens=len(prompt_ids), cap=cap, snap_before=snap_before,
                tokens=row["tokens"], finish=row["finish"],
                hits_before=hits_before, error=row["error"])

    async def stream(self, req, cancel=None, prepared=None):
        """Async generator yielding ("delta", text) then ("final", GenerationResult).

        Bridges vaayu.streaming's worker thread onto the event loop through a
        thread-safe queue.

        `prepared` is `prepare(req)`'s result, already computed by the caller.
        The HTTP route computes it BEFORE returning its StreamingResponse so
        that ContextTooLong and friends can still become a 400 (B8); accepting
        it here is what keeps that from costing a second render + tokenize pass.
        """
        import time

        from ..streaming import stream_generate

        if cancel is None:
            # Always publish a REAL token. `cancel_all()` reads it to answer
            # "is anything generating?", so a None there tells the SIGTERM
            # handler and `_teardown` the engine is idle while the GPU is busy
            # -- and the teardown below would then have nothing to cancel
            # before it waits for the worker (B9).
            cancel = CancelToken()

        started, hits_before = time.time(), self._prefix_hits()
        snap_before = reqlog.snapshot(self.model)
        row = {"tokens": 0, "finish": "error", "error": None, "ttft_s": None}
        # Defaults for the `finally` below, in case `prepare()` itself raises
        # (the fallback path only -- the HTTP route normally passes `prepared`
        # already computed, per the docstring above): the row must still be
        # buildable rather than raising a NameError that replaces the real
        # exception on its way out.
        prompt_ids, cap = [], 0
        try:
            prompt_ids, cap = prepared if prepared is not None else self.prepare(req)
            self._log(req, "REQ", prompt=len(prompt_ids), cap=cap, stream=True,
                      thinking=("on" if self.resolve_thinking(req) else "off"))
            ttft_logged = False
            kwargs = self.build_generate_kwargs(req, prompt_ids, cap, cancel=cancel)
            input_ids = kwargs.pop("input_ids")
            temperature = kwargs.get("temperature", 0.0)
            top_p = kwargs.get("top_p", 1.0)
            # stream_generate builds these itself; leaving any of them in **kwargs is
            # a duplicate-keyword TypeError, which is how `use_cache` shipped broken
            # past every fake-backed test and was caught only on-device.
            for k in STREAM_GENERATE_OWNS:
                kwargs.pop(k, None)

            loop = asyncio.get_running_loop()
            bridge: _queue.Queue = _queue.Queue()
            DONE = object()
            # The real generated ids, filled by the streamer as they are produced.
            # Re-tokenizing the decoded text does not reproduce them -- it is decoded
            # with skip_special_tokens=True, so EOS is gone and merges differ across
            # a decode boundary -- and both completion_tokens and finish_reason are
            # derived from the count.
            new_ids: list = []

            def _pump():
                try:
                    for chunk in stream_generate(self.model, input_ids,
                                                 max_new_tokens=cap,
                                                 temperature=temperature,
                                                 top_p=top_p,
                                                 token_ids_out=new_ids, **kwargs):
                        bridge.put(("delta", chunk))
                except BaseException as e:              # noqa: BLE001 -- re-raised below
                    bridge.put(("error", e))
                finally:
                    bridge.put((DONE, None))

            thread = threading.Thread(target=_pump, daemon=True)
            self._live_cancel = cancel
            self._live_thread = thread
            thread.start()

            parts, error = [], None
            try:
                while True:
                    kind, payload = await loop.run_in_executor(None, bridge.get)
                    if kind is DONE:
                        break
                    if kind == "error":
                        error = payload
                        break
                    parts.append(payload)
                    if not ttft_logged:
                        ttft_logged = True
                        row["ttft_s"] = time.time() - started
                        self._log(req, "TTFT", at=f"{row['ttft_s']:.1f}s")
                    yield ("delta", payload)
            except BaseException:                   # noqa: BLE001 -- re-raised
                # The consumer went away -- Starlette closes this generator with
                # GeneratorExit, which is a BaseException -- or threw in. The
                # worker is still inside model.generate, so cancel it before the
                # `finally` waits: an uncancelled join here would block for the
                # whole remaining cap.
                exc = sys.exc_info()[1]
                if isinstance(exc, GeneratorExit):
                    # A disconnect is not an error, and `finish` must say so
                    # too: it is still at its "error" default here (the
                    # success line below is never reached on this path), and
                    # a row reading finish="error", error=null is incoherent
                    # to a maintainer reading it cold.
                    row["error"] = None
                    row["finish"] = "cancelled"
                else:
                    row["error"] = exc
                cancel.cancel()
                raise
            finally:
                # The token stays published until the worker has ACTUALLY exited.
                # Clearing it first -- which this `finally` used to do, on a path
                # that also skipped the join entirely -- makes `cancel_all()`
                # report idle with a generation still on the GPU, so a SIGTERM
                # shutdown proceeds underneath it (B9). It is also the token
                # `slot()`'s `quiesce()` cancels through (B4).
                try:
                    thread.join()
                finally:
                    self._live_cancel = None
                    self._live_thread = None
            if error is not None:
                row["error"] = error
                raise error

            full = "".join(parts)
            cleaned, calls = parse_tool_calls(full, self.tool_format if req.tools else None,
                                      declared_tool_names(req.tools))
            produced = len(new_ids)
            finish = self._finish(new_ids, cap, calls, cancel=cancel)
            wall = time.time() - started
            self._log(req, "DONE", tokens=produced, finish=finish,
                      prefix_hit=self._prefix_hits() > hits_before, wall=f"{wall:.1f}s",
                      decode=f"{(produced - 1) / wall:.3f}tok/s" if wall > 0 and produced > 1 else "n/a")
            row["tokens"], row["finish"] = produced, finish
            yield ("final", GenerationResult(
                text=cleaned, tool_calls=calls,
                prompt_tokens=len(prompt_ids), completion_tokens=produced,
                finish_reason=finish))
        except BaseException as ex:              # noqa: BLE001 -- re-raised
            # An OUTER net, not a replacement for the inner handler above:
            # that one still cancels the in-flight generation before
            # re-raising, and an uncancelled join would block for the whole
            # remaining token budget -- this one does not touch that. It
            # exists because setup (build_generate_kwargs, thread creation)
            # and finalize (parse_tool_calls, _finish) both run OUTSIDE the
            # inner consumer-loop try, so a failure there used to reach the
            # `finally` below with row["error"] still None -- a real
            # traceback logged as a clean request, exactly the defect this
            # file exists to eliminate. Idempotent when the inner handler
            # already ran: same exception, same GeneratorExit check.
            if isinstance(ex, GeneratorExit):
                row["error"] = None
                if row["finish"] == "error":
                    # Only overwrite the default -- a GeneratorExit landing
                    # HERE after the success line already ran (the client
                    # vanished right at the final yield) must not clobber a
                    # real finish_reason with "cancelled".
                    row["finish"] = "cancelled"
            else:
                row["error"] = ex
            raise
        finally:
            self._write_request_row(
                req, stream=True, started=started,
                prompt_tokens=len(prompt_ids), cap=cap, snap_before=snap_before,
                tokens=row["tokens"], finish=row["finish"], ttft_s=row["ttft_s"],
                hits_before=hits_before, error=row["error"])
