"""EngineHandle -- the one loaded model, request mapping, and generation.

The engine is not thread-safe across the KV cache, the expert VRAM cache and
the slab arena, so exactly one generation runs at a time -- but since P10 that
generation may be a BATCH of one or N. `vaayu/server/scheduler.py` owns the
dispatch loop and holds `self._lock` for a whole run; this file provides the
two things it can run: `stream()` (one request, unchanged) and `run_batch()`
(N requests through `batched_generate`).

This docstring used to say N>=2 at long context was "a settled hardware
refusal on this box" and that `batched_generate` "would admit nothing here".
Both were measured false on 2026-09-07/08: at ~4.15k-token agentic prompts
admission grants N=2 on MiniMax-M2.7 (+1.606x end-to-end) after the P10a
expert-cache giveback. Qwen3-235B still admits N=1 only -- its cache budget is
already below one decode step's working set, so it has nothing to surrender --
and N=1 is served serially, never through `batched_generate`, which is worth
3.24x to it.
"""
import asyncio
import queue as _queue
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import reqlog
from .chat import NoChatTemplate, TemplateLacksToolSupport, render_prompt
from .logs import clear_current_session, session_event, set_current_session
from .stopping import ContextTooLong, finish_reason, resolve_max_tokens
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


class _Detok:
    """Incremental per-stream detokenization for the batched path.

    Decoding one id at a time does NOT reproduce the text -- merges differ
    across a decode boundary -- so the whole id list is decoded each time and
    only the delta is emitted, which is what TextIteratorStreamer does. A
    trailing replacement char means the last id was half of a multi-byte
    character: hold it back rather than sending U+FFFD to the client and then
    contradicting it on the next token.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.text = ""

    def feed(self, token_id: int) -> str:
        self.ids.append(token_id)
        full = self.tok.decode(self.ids, skip_special_tokens=True)
        if full.endswith("�"):
            return ""
        delta, self.text = full[len(self.text):], full
        return delta


class _BatchStream:
    """One stream inside a batch: its ids, its text, its stop policy.

    Stop STRINGS, not ids. `BatchRequest.stop_ids` is token ids while the
    client's `stop` and stopping.py are strings, and tokenizing a stop string
    to bridge them is wrong at merge boundaries -- the same string tokenizes
    differently depending on what precedes it. Matching therefore happens here,
    on decoded text, and trims AT the match exactly as HF does on the serial
    path: the same request must not return different text because it happened
    to be batched.

    `retired` is what the engine's should_stop hook reads. It carries the
    stop-string match and the per-stream cap; the caller ORs in its own
    cancellation.
    """

    def __init__(self, job, tokenizer, stops, cap: int):
        self.job = job
        self.stops = list(stops or [])
        self.cap = cap
        self._detok = _Detok(tokenizer)
        self.text = ""
        self.retired = False

    @property
    def ids(self):
        return self._detok.ids

    def feed(self, token_id: int) -> str:
        """Consume one token. Returns the delta to SEND -- "" for nothing."""
        if self.retired:
            return ""
        delta = self._detok.feed(token_id)
        full = self._detok.text
        cut = self._earliest_stop(full)
        if cut is not None:
            # Trim at the stop and retract anything already sent past it: the
            # final text is what HF would have returned, not what the deltas
            # happened to add up to.
            self.text = full[:cut]
            self.retired = True
            keep = len(self.text) - (len(full) - len(delta))
            return delta[:keep] if keep > 0 else ""
        self.text = full
        if len(self.ids) >= self.cap:
            self.retired = True
        return delta

    def _earliest_stop(self, full: str):
        cuts = [full.index(s) for s in self.stops if s in full]
        return min(cuts) if cuts else None


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
        # The tokens of the requests currently generating, so shutdown can stop
        # them. A LIST since P10: the dispatcher runs a batch of one OR N, and
        # cancel_all() -- which SIGTERM and _teardown reach the GPU through --
        # must reach every member, not whichever one was published last.
        self._live_cancels = []
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
    def live_cancels(self):
        """Every running request's CancelToken. Empty when idle."""
        return list(self._live_cancels)

    @property
    def live_cancel(self):
        """The first running request's CancelToken, or None when idle.

        Kept for callers that only ask "is anything generating?". Under a batch
        it is one member of several -- use `live_cancels` to act on them all.
        """
        return self._live_cancels[0] if self._live_cancels else None

    def cancel_all(self) -> None:
        """Stop whatever is generating. Idempotent, and a no-op when idle."""
        for token in list(self._live_cancels):
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
                           ttft_s=None, hits_before: int = 0, error=None,
                           batch=None) -> None:
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
            # Under N>1: `tokens` and `ttft_s` are genuinely per-stream. But
            # `wall_s` and `decode_tok_s` are NOT -- `started` is the whole
            # batch's start, so a stream that retires early is still charged
            # the batch's full wall time and its decode_tok_s understates.
            # `prefix_hit` is batch-wide too (it compares process counters),
            # so it reads True on every row if any one stream hit. `decode`
            # (below) is the batch's bytes and stalls, not this stream's --
            # flagged `shared=True` so nothing sums N rows into a number that
            # never happened.
            if batch:
                decode = dict(reqlog.decode_block(snap_before, snap_after, tokens),
                              shared=True)
            else:
                decode = reqlog.decode_block(snap_before, snap_after, tokens)
            reqlog.write_row(self.log_dir, {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "session": getattr(req, "session_key", None),
                "batch": batch,
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
                "decode": decode,
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

    def notice_if_speculation_skipped(self, req):
        """speculative_generate is GREEDY -- temperature included -- and has no
        stop-string machinery. Return a message when this request asks for
        something that path would drop, else None.

        Nothing is LOST when this fires: `stream_generate` dispatches per
        request, so such a request takes the ordinary path and gets what it
        asked for. The message says which control sent it there, so a caller
        wondering why speculation did not engage has the answer. It is asked of
        the same helper that dispatch uses, against the kwargs this request
        would actually be run with, so the two can never disagree.
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
        return (f"speculative decoding is SKIPPED for this request: its loop "
                f"is greedy and has no stop-string machinery, so it would drop "
                f"{', '.join(ignored)}. The request runs on the ordinary path, "
                f"which honours them.")

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
        self._live_cancels = [cancel]
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
                self._live_cancels = []
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

    async def generate(self, req, cancel=None, prepared=None) -> GenerationResult:
        import time
        if cancel is None:
            # Always publish a REAL token, for the same reason stream() does
            # (B9): `cancel_all()` reads it to answer "is anything generating?",
            # and the teardown below has to have something to cancel before it
            # waits for the worker (B16).
            cancel = CancelToken()
        started, hits_before = time.time(), self._prefix_hits()
        snap_before = reqlog.snapshot(self.model)
        row = {"tokens": 0, "finish": "error", "error": None}
        # Same defaults, and for the same reason, as stream()'s below: prepare()
        # is INSIDE the try so that a failure there still reaches the row-writing
        # `finally`. It raises more than the three types the route names --
        # apply_chat_template runs Jinja over the messages, and a malformed
        # sequence (a `tool` message with no matching call, an unexpected role
        # order) raises whatever the template raises. That used to skip the row
        # entirely on both paths, against this file's one contract.
        #
        # `prepared` is `prepare(req)`'s result, already computed by the
        # caller -- same contract as stream()'s: accepting it here is what
        # keeps a scheduler-dispatched non-streaming job from re-rendering the
        # chat template and re-tokenizing a prompt it already tokenized.
        prompt_ids, cap = [], 0
        try:
            prompt_ids, cap = prepared if prepared is not None else self.prepare(req)
            self._log(req, "REQ", prompt=len(prompt_ids), cap=cap, stream=False,
                      thinking=("on" if self.resolve_thinking(req) else "off"))
            if req.max_tokens is None:
                # Non-streaming sends NOTHING until the last token, so this
                # request can sit silent for as long as `cap` tokens take --
                # measured at 15+ minutes and counting on MiniMax-M2.7 before a
                # client gave up and the run looked hung. It is not hung; it is
                # uncapped.
                print(f"[Vaayu-Server] WARNING: non-streaming request with no "
                      f"max_tokens: up to {cap} tokens will be generated before "
                      f"ANY bytes are sent. Set max_tokens, or use stream=true.")
            kwargs = self.build_generate_kwargs(req, prompt_ids, cap, cancel=cancel)
            loop = asyncio.get_running_loop()
            # An OWN thread, not a run_in_executor pool worker: `quiesce()` joins
            # what is published here, and a pool thread never exits, so joining one
            # would hang forever. The bridge mirrors stream()'s pump.
            bridge: _queue.Queue = _queue.Queue()

            # Per-request, exactly as the streaming route decides it: this
            # route called `model.generate` unconditionally, so a speculator was
            # never used here at all -- measured 2026-09-06 as 10 draft rounds
            # streamed against 0 non-streamed, same greedy prompt, same process.
            from ..vaayu_base import choose_speculative_target
            target = choose_speculative_target(
                self.model, kwargs.get("temperature", 0.0), kwargs)

            def _work():
                try:
                    bridge.put(("out", target(**kwargs)))
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
            if cancel.cancelled:
                # `finish_reason` maps cancelled -> "length" on purpose: the
                # OpenAI wire vocabulary has no other word for it. requests.jsonl
                # is NOT the wire, and documentation/server.md's table promises
                # `cancelled` there. Without this, a non-streaming disconnect is
                # indistinguishable from a truncated answer. The wire value
                # returned below is untouched -- stream() already does the same.
                row["finish"] = "cancelled"
            return GenerationResult(
                text=cleaned, tool_calls=calls,
                prompt_tokens=len(prompt_ids), completion_tokens=len(new_ids),
                finish_reason=finish)
        except BaseException as ex:              # noqa: BLE001 -- re-raised
            row["error"] = ex
            if isinstance(ex, (ContextTooLong, TemplateLacksToolSupport,
                               NoChatTemplate)):
                # Rejected before generation started, so the row says `refused`
                # -- the vocabulary documentation/server.md promises, and what
                # the route's own `_refusal_row` wrote for these three before
                # prepare() moved inside this try. The route no longer writes
                # one for them; that would now be two rows for one request.
                row["finish"] = "refused"
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
            self._live_cancels = [cancel]
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
                    self._live_cancels = []
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
                if isinstance(ex, (ContextTooLong, TemplateLacksToolSupport,
                                   NoChatTemplate)):
                    # Same word generate() uses for these three: rejected
                    # before generation started. Only the fallback prepare()
                    # above can raise them here -- the HTTP route passes
                    # `prepared`, so this is for other callers of stream() --
                    # but the vocabulary must not depend on which method you
                    # called any more than on which path you took (V1).
                    row["finish"] = "refused"
            raise
        finally:
            self._write_request_row(
                req, stream=True, started=started,
                prompt_tokens=len(prompt_ids), cap=cap, snap_before=snap_before,
                tokens=row["tokens"], finish=row["finish"], ttft_s=row["ttft_s"],
                hits_before=hits_before, error=row["error"])

    async def run_batch(self, jobs):
        """Drive ONE batched_generate over `jobs`, fanning tokens to each queue.

        THE GENERATOR CONTRACT IS THIS METHOD'S WHOLE REASON FOR EXISTING.
        `batched_generate`'s docstring: abandoning it mid-iteration leaves the
        model in batched-decode mode (`_batch_decode_active`/`_batch_n`) until
        Python garbage-collects the generator -- and a client disconnect is
        exactly that abandonment. Here the generator is owned by ONE worker
        thread that always closes it, and clients hold queues rather than
        generators, so no client CAN abandon it. A disconnect retires that row
        through should_stop and the other streams keep running.

        Each job receives ("delta", text)*, then one ("final", result) or
        ("error", exc), then None as its end sentinel. Every job gets a
        sentinel on every path -- EXCEPT one: if `batched_generate` fails
        before ANY job received a real delta (a gate-2 admission refusal,
        typically -- it is a generator function, so nothing runs until the
        first `next()`), this raises the underlying error instead of fanning
        it out, and no job's queue has anything in it. That is deliberate:
        the caller (BatchScheduler._run) can then retry the same jobs
        serially without any risk of a client seeing output twice. A failure
        that happens AFTER some job already got a real delta still fans out
        to every job exactly as before -- a client waits forever on a queue
        nobody will write to again. (A real cancellation racing with that
        raise is not swallowed by it either -- see `loop_raised` below.)

        On the raise path, NO requests.jsonl row is written for any of
        `jobs` here -- `_write_request_row` lives inside the per-job
        fan-out loop this raise skips. That is correct, not a gap: the
        caller's serial retry runs each job through stream()/generate(),
        which write their own row exactly once each. Fanning out AND
        raising would double-count every retried request.
        """
        import time

        from ..batch import BatchRequest

        loop = asyncio.get_running_loop()
        started = time.time()
        hits_before = self._prefix_hits()
        snap_before = reqlog.snapshot(self.model)
        n = len(jobs)
        members = [getattr(j, "session_key", None) for j in jobs]
        batch_note = {"n": n, "members": members}

        streams, reqs = [], []
        for j in jobs:
            kwargs = self.build_generate_kwargs(j.req, j.prompt_ids, j.cap,
                                                cancel=j.cancel)
            stops = list(kwargs.get("stop_strings") or [])
            streams.append(_BatchStream(j, self.model.tokenizer, stops, j.cap))
            reqs.append(BatchRequest(
                prompt_ids=list(j.prompt_ids), max_new_tokens=j.cap,
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 1.0),
                # Empty on purpose: EOS is unioned inside batched_generate, and
                # the client's stops are STRINGS, matched by _BatchStream.
                stop_ids=()))
            self._log(j.req, "REQ", prompt=len(j.prompt_ids), cap=j.cap,
                      stream=True, batch=n)

        bridge: _queue.Queue = _queue.Queue()
        DONE = object()

        def _should_stop(i):
            return streams[i].retired or jobs[i].cancel.cancelled

        def _pump():
            # `batched_generate(...)` itself is INSIDE the try: real
            # implementations are plain generator functions (nothing runs
            # until the first `next()`), so this is theoretical today, but a
            # model whose `batched_generate` is a wrapper that does work on
            # call would otherwise raise before `gen` exists at all, killing
            # this thread with no error/DONE ever reaching the bridge and the
            # awaiting coroutine stuck on `bridge.get()` forever.
            gen = None
            try:
                gen = self.model.batched_generate(reqs, should_stop=_should_stop)
                for stream_id, tok in gen:
                    bridge.put(("tok", (stream_id, tok)))
            except BaseException as e:           # noqa: BLE001 -- re-raised below
                bridge.put(("error", e))
            finally:
                # Explicit, never GC: this is the contract in the docstring
                # above. A no-op when the loop finished normally. `gen` is
                # None only if the call above itself raised, in which case
                # there is nothing to close. The DONE put is its own nested
                # finally -- gen.close() runs the model's substantial
                # teardown (expert-cache budget restore, VRAM prefix
                # restore, flag clearing), and if THAT raises, the sentinel
                # must still reach the bridge or the awaiting coroutine
                # blocks on bridge.get() forever, holding the lock.
                try:
                    if gen is not None:
                        gen.close()
                finally:
                    bridge.put((DONE, None))

        thread = threading.Thread(target=_pump, daemon=True)
        ttft = [None] * n
        error = None
        # Published BEFORE `start()`, matching `stream()`/`_running`: a
        # SIGINT/SIGTERM handler runs synchronously on the main thread
        # between bytecodes (app.py installs it via `signal.signal`), so a
        # signal landing between `start()` and a later publish would find
        # `_live_cancels` still empty and `cancel_all()` would silently lose
        # the cancel -- `cancel_all`'s own docstring says lifespan teardown
        # does not fire early enough to compensate, so the cost is the
        # "still alive 20s after SIGTERM" failure it exists to prevent.
        #
        # `start()` is wrapped rather than left inside a bare try above the
        # loop below: if it raises (e.g. "can't start a new thread"), this
        # publish must not be left standing -- `live_cancel` would then
        # report a generation in flight forever, and `quiesce()` (the
        # teardown / client-disconnect path) would `.join()` a Thread that
        # was never started, hitting "cannot join before started". So the
        # except resets both to idle before re-raising.
        self._live_cancels = [j.cancel for j in jobs]
        self._live_thread = thread
        try:
            thread.start()
        except BaseException:
            self._live_cancels = []
            self._live_thread = None
            raise
        # Whether ANY job's real .out queue has received a client-visible
        # item yet. Distinguishes "refused before producing anything" (a
        # gate-2 admission refusal inside batched_generate, raised before it
        # ever yields a token) from "failed after real output was already
        # delivered" -- see the re-raise below.
        delivered_any = False
        loop_raised = False
        try:
            while True:
                kind, payload = await loop.run_in_executor(None, bridge.get)
                if kind is DONE:
                    break
                if kind == "error":
                    error = payload
                    continue
                i, tok = payload
                delta = streams[i].feed(tok)
                if delta:
                    if ttft[i] is None:
                        ttft[i] = time.time() - started
                        self._log(jobs[i].req, "TTFT", at=f"{ttft[i]:.1f}s")
                    await jobs[i].out.put(("delta", delta))
                    delivered_any = True
        except BaseException:
            # Recorded HERE, in this frame's own handler, rather than read
            # back later via sys.exc_info() in the finally below: exc_info()
            # reflects the whole THREAD's currently-handled exception, not
            # this method's own try -- if run_batch is ever awaited from
            # inside a CALLER's own except/finally, an ambient check would
            # see the CALLER's exception (verified: it does, across an
            # await, in CPython) and wrongly suppress the raise below even
            # though nothing in this loop actually failed. A local flag,
            # set only when THIS try's own body raised, cannot be fooled by
            # an outer context this method never sees.
            loop_raised = True
            raise
        finally:
            try:
                thread.join()
            finally:
                self._live_cancels = []
                self._live_thread = None
            if error is not None and not delivered_any and not loop_raised:
                # Nothing reached a real client queue before the failure, so
                # there is nothing here for a caller to double-deliver: raise
                # instead of fanning the error out to every job. This is what
                # lets BatchScheduler._run tell "safe to retry serially" from
                # "some client already has real output, retrying would risk
                # sending it twice" without a caller-side peek/buffer of its
                # own -- see vaayu/server/scheduler.py's `_run`.
                #
                # `not loop_raised` guards against a DIFFERENT exception
                # already unwinding through this finally -- a CancelledError
                # delivered while awaiting the next bridge item,
                # realistically, once `error` was already set from an
                # earlier one. Without the guard, `raise error` here would
                # REPLACE that CancelledError with `error` (a plain
                # Exception), and BatchScheduler._run's `except Exception`
                # would then read a real cancellation as an ordinary
                # formation refusal and run N serial generations inside a
                # task that was asked to stop -- the opposite of what
                # cancellation means. Falling through in that case lets the
                # original exception keep propagating; the per-job fan-out
                # loop below still runs first (it always does, on every
                # path, via this same `finally`), so no client is left with
                # nothing in its queue.
                raise error
            # A BaseException that is NOT an Exception (CancelledError,
            # KeyboardInterrupt) raised while finalising one job must still
            # propagate -- swallowing it as a client-visible error would
            # silently drop a cancellation. But it must not starve i+1..N
            # either, so it is recorded here and re-raised only once every
            # job has its result/error item and its sentinel.
            pending_base_exc = None
            for i, j in enumerate(jobs):
                row = {"tokens": len(streams[i].ids), "finish": "error",
                       "error": error}
                try:
                    try:
                        if error is not None:
                            await j.out.put(("error", error))
                        else:
                            cleaned, calls = parse_tool_calls(
                                streams[i].text,
                                self.tool_format if j.req.tools else None,
                                declared_tool_names(j.req.tools))
                            finish = self._finish(streams[i].ids, j.cap, calls,
                                                  cancel=j.cancel)
                            row["finish"] = finish
                            if j.cancel.cancelled:
                                # Same rule generate() applies: requests.jsonl is
                                # not the wire, and documentation/server.md promises
                                # `cancelled` there.
                                row["finish"] = "cancelled"
                            self._log(j.req, "DONE", tokens=len(streams[i].ids),
                                      finish=finish, batch=n)
                            await j.out.put(("final", GenerationResult(
                                text=cleaned, tool_calls=calls,
                                prompt_tokens=len(j.prompt_ids),
                                completion_tokens=len(streams[i].ids),
                                finish_reason=finish)))
                    except Exception as e:
                        # A failure finalising job i (parse_tool_calls,
                        # _finish, _log, or building GenerationResult) must
                        # not propagate out of this loop: that would skip
                        # jobs i+1..N entirely, leaving their clients blocked
                        # on out.get() forever with no ("final")/("error")
                        # item and no sentinel.
                        row["error"] = e
                        await j.out.put(("error", e))
                    except BaseException as e:  # noqa: BLE001 -- re-raised below
                        row["error"] = e
                        await j.out.put(("error", e))
                        if pending_base_exc is None:
                            pending_base_exc = e
                finally:
                    # The sentinel on EVERY path, or a client waits forever on
                    # a queue nobody will write to again.
                    await j.out.put(None)
                    self._write_request_row(
                        j.req, stream=True, started=started,
                        prompt_tokens=len(j.prompt_ids), cap=j.cap,
                        snap_before=snap_before, tokens=row["tokens"],
                        finish=row["finish"], ttft_s=ttft[i],
                        hits_before=hits_before, error=row["error"],
                        batch=batch_note)
            if pending_base_exc is not None:
                # Every job has its item and its sentinel by now (the loop
                # above ran to completion regardless of this exception) --
                # only now does cancellation propagate, as asyncio requires.
                raise pending_base_exc
